import h5py 
import numpy as np
from rich.progress import track
import time
from contextlib import contextmanager
from omninet.dataset.types import IndexDict

def print_structure(name, obj, structure = dict()):
    """
    Function to print the structure of the HDF5 file.
    It prints the name and type of each object (group or dataset).
    """
    if isinstance(obj, h5py.Group):
        for subname, subobj in obj.items():
            print_structure(f"{name}/{subname}", subobj, structure)
    elif isinstance(obj, h5py.Dataset):
        structure[name] = {"shape": obj.shape, "dtype": obj.dtype}

    return structure

def explore_hdf5(file_path, structure = dict()):
    """
    Function to open an HDF5 file and print its structure.
    """
    try:
        with h5py.File(file_path, 'r') as file:
            for name, obj in file.items():
                structure = print_structure(name, obj, structure)
        return structure
    except Exception as e:
        print(f"Error opening file: {e}")

class VirtualHDF5Dataset():
    def __init__(self, filepaths, dataset_name, nevents_summary, shape, dtype):
        """
        Initializes a VirtualHDF5Dataset that combines datasets across multiple HDF5 files.
       
        Args:
            filepaths (list of str): List of file paths to HDF5 files.
            dataset_name (str): Name of the dataset to combine from each file.
            nevents_summary (dict[str]): Dict of number of events for each file.
            shape: shape
            dtype: np.dtype
        """

        self.filepaths = filepaths
        self.dataset_name = dataset_name

        self.total_size = 0
        self.nevents_summary = nevents_summary
        self.shape   = tuple(shape)

        self.dtype = dtype
        self.dataset_shapes = []

        for filepath in self.filepaths:
            dataset_shape = (nevents_summary[filepath],) + self.shape
            self.dataset_shapes.append(dataset_shape) 
            self.total_size += nevents_summary[filepath]
        self.cumulative_offsets = np.cumsum([0] + [shape[0] for shape in self.dataset_shapes])
        self.shape = (self.total_size,) + self.shape
        self.files = None


    def find_file_index(self, indices):

        file_indices = np.searchsorted(self.cumulative_offsets, indices, side='right') - 1
        local_index  = np.arange(len(indices))
        array_dict   = dict()
        unique_file_indices = np.unique(file_indices)
        nEvent = len(indices)
        indices = np.array(indices)
        for file_index in unique_file_indices:
            mask = (file_indices == file_index)
            array_dict[file_index] = (local_index[mask], indices[mask] - self.cumulative_offsets[file_index])
        return IndexDict(array_dict, nEvent)

    def set_files(self, files):
        self.files = files

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        """
        Retrieve an item or a slice of items from the virtual dataset.

        Args:
            index(int or slice): The index or slice to retrieve.
     
        Returns:
            numpy.ndarray: The data corresponding to the requested index or slice.

        """
        if isinstance(index, int) or isinstance(index, np.int64) or isinstance(index, np.int32):
            # Handle negative indexing
            if index < 0:
                index += self.total_size
            if index < 0 or index >= self.total_size:
                raise IndexError("Index out of range.")

            # Find the corresponding file and index within taht file
            file_idx = np.searchsorted(self.cumulative_offsets, index, side='right') - 1
            local_index = index - self.cumulative_offsets[file_idx]
 
            f = self.files[file_idx]
            if self.dataset_name in f:
                return f[self.dataset_name][local_index]
            else:
                if self.dtype in [np.float32]:
                    return -1.0
                elif self.dtype in [np.int64]:
                    return -1
                else:
                    return False

        elif isinstance(index, slice):
            # Handle slices
            start, stop, step = index.indices(self.total_size)
            if step != 1:
                raise ValueError("Step size other than 1 is not supported.")

            return self[np.arange(start, stop, step)]

        elif isinstance(index, list) or isinstance(index, tuple) or isinstance(index, np.ndarray):
            index = np.array(index)
            array_dict = self.find_file_index(index)
            return self[array_dict]

        elif isinstance(index, IndexDict):
            array_dict = index
            output_array = np.empty((len(index), ) + self.shape[1:], dtype = self.dtype)

            missing_indices = []

            for file_index, (dest_sel, source_sel) in array_dict.items():
                f = self.files[file_index]
                if self.dataset_name in f:
                    f[self.dataset_name].read_direct(output_array, source_sel = source_sel, dest_sel = dest_sel)
                else:
                    if self.dtype in [np.float32, np.int64]:
                        output_array[dest_sel] = np.full((len(dest_sel),) + self.shape[1:], -1, dtype = self.dtype)
                    else:
                        output_array[dest_sel] = np.full((len(dest_sel),) + self.shape[1:], False, dtype = self.dtype)


            return output_array

        else:
            raise TypeError(f"Index must be an integer or a slice. type = {type(index)}")

    def read_direct(self, destination, source_sel=None, dest_sel = None):
        """
        Custom implementation of read_direct to read data from multiple files 
        and write it directly into the destination array.
        :param destination: NumPy array to store the data
        :param source_sel: Optional slice or index to read specific data from the source
        """
        if not isinstance(destination, np.ndarray):
            raise TypeError("Destination must be a NumPy array")


        # If source_sel is provided, apply it; otherwise, read the entire dataset
        if dest_sel is not None:
            if source_sel is not None:
                destination[dest_sel] = self[source_sel]
            else:
                destination[dest_sel] = self[:]  # Read the whole dataset

        else:
            if source_sel is not None:
                destination[:] = self[source_sel]
            else:
                destination[:] = self[:]  # Read the whole dataset


class  VirtualHDF5:
    def __init__(self, filepaths):
        """
        Initializes a VirtualHDF5 that combines dataset across multiple HDF5 fies.
        
        Args:
            filepaths (list of str): List of file paths to HDF5 files.
        """

        self.nevent_summary = dict()
        self.shape_summary  = dict()
        self.dtype_summary  = dict()
        self.file_summary   = dict()
       
        self.total_size     = 0

        self.filepaths      = filepaths
        self.files          = [None for file_ in self.filepaths]
        self.dataset        = dict()

        self.dataset_shapes = []

        for fname in track(self.filepaths, description = "Intialize h5py dataset..."):
            structure = explore_hdf5(fname, structure = dict())
            nevent    = None
            for dataset_name in structure:
                shape = structure[dataset_name]["shape"]  
                dtype = structure[dataset_name]["dtype"]
                shape_tail = [] if len(shape) == 1 else shape[1:]

                if nevent is None:
                    nevent = shape[0]
                else:
                    assert nevent == shape[0], f"nEvents are not consistent in {fname}, original = {nevent}, {dataset_name} = {shape[0]}"
                if dataset_name not in self.shape_summary:
                    self.shape_summary[dataset_name] = shape_tail
                    self.dtype_summary[dataset_name] = dtype
                else:
                    assert self.shape_summary[dataset_name] == shape_tail, f"shapes are not consistent in {dataset_name}, 1 = {self.shape_summary[dataset_name]}, 2 = {shape_tail}"
                    assert self.dtype_summary[dataset_name] == dtype, f"dtypes are not consistent in {dataset_name}, 1 = {self.dtype_summary[dataset_name]}, 2 = {dtype}"

            self.nevent_summary[fname] = nevent
            self.total_size += nevent
            self.dataset_shapes.append([nevent])

        for dataset_name in self.shape_summary:
            vshape = (self.total_size, ) + tuple(self.shape_summary[dataset_name])
            dtype  = self.dtype_summary[dataset_name] 
#            self.dataset[dataset_name] = self.create_virtual_dataset(vshape, dtype, dataset_name)
            self.dataset[dataset_name] = VirtualHDF5Dataset(self.filepaths, dataset_name, self.nevent_summary, self.shape_summary[dataset_name], self.dtype_summary[dataset_name])
        self.cumulative_offsets = np.cumsum([0] + [shape[0] for shape in self.dataset_shapes])



    def create_virtual_dataset(self, vshape, dtype, dataset_name):
        layout = h5py.VirtualLayout(shape = vshape, dtype = dtype) 

        start_idx = 0
        for fname in self.filepaths:
            nevent = self.nevent_summary[fname]
            with h5py.File(fname, 'r') as f:
                if dataset_name in f:
                    vsource = h5py.VirtualSource(fname, dataset_name, shape=(nevent,) + tuple(self.shape_summary[dataset_name]))
                    layout[start_idx:start_idx + nevent] = vsource
                else:
                    if self.dtype_summary[dataset_name] in [np.float32, np.int64]:
                        layout[start_idx:start_idx + nevent] = -1
                    else:
                        layout[start_idx:start_idx + nevent] = 0
            start_idx = start_idx + nevent
        return layout

    # Context manager for opening and closing multiple HDF5 files
    @contextmanager
    def open(self):

        try:
            for path_idx, path in enumerate(self.filepaths):
                self.files[path_idx] = h5py.File(path, 'r', swmr=True)  # Open the files in read mode
                for dataset_ in self.dataset:
                    self.dataset[dataset_].set_files(self.files)
            yield self  # Yield the opened files for use in the context

        finally:
            # Close all files after the context block is finished
            for file_ in self.files:
                file_.close()
    

    def open_files(self):
        for path_idx, path in enumerate(self.filepaths):
            self.files[path_idx] = h5py.File(path, 'r', swmr=True)
            for dataset_ in self.dataset:
                self.dataset[dataset_].set_files(self.files)
        return self

    def close_files(self):
        for file_ in self.files:
            file_.close()


    def find_file_index(self, indices):

        file_indices = np.searchsorted(self.cumulative_offsets, indices, side='right') - 1
        local_index  = np.arange(len(indices))
        array_dict   = dict()
        unique_file_indices = np.unique(file_indices)
        nEvent = len(indices)
        indices = np.array(indices)
        for file_index in unique_file_indices:
            mask = (file_indices == file_index)
            array_dict[file_index] = (local_index[mask], indices[mask] - self.cumulative_offsets[file_index])
        return IndexDict(array_dict, nEvent)

           
    def __getitem__(self, key):
        """
        Retrieve an item for given key
        """

        if key in self.dataset:
            return self.dataset[key]
        else:
            raise KeyError(f"{key} not in the dataset")

    def __iter__(self):
        return iter(self.dataset.keys())
