/**

 MPI_File_read_all minimal reproducer
 
 Source file: MPIRead32.cpp
 
 Copyright (C) 2022
 
 Author: Michael Marshall (mmphys)
 Git: git@github.com:mmphys/MPIRead32.git
 
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License along
 with this program; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 
 See the full license in the file "LICENSE" in the top level distribution directory
**/

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// Forward declarations for plumbing (implementations at end of file)
template <class _CharT, class _Traits> inline bool StreamEmpty( std::basic_istream<_CharT, _Traits> & s );
template<typename T> inline T FromString( const std::string &String );
template<typename T> inline std::vector<T> ArrayFromString( const std::string &String );
template<typename T> std::ostream & operator<<( std::ostream &s, const std::vector<T> &v );

struct Reproducer
{
  // Instantiate a Reproducer, then once (or more) set MPIDims + GlobalDims and call Test
  void Test( const char * FileName, long HeaderBytes );

  const int world_size;
  const int world_rank;
  std::vector<int> MPIDims;
  std::vector<int> GlobalDims;

  Reproducer() : world_size{getWorldSize()}, world_rank{getWorldRank()} {}
  ~Reproducer() { MPI_Finalize(); }

protected:
  // I'll use a block of doubles of the same size as a Lattice QCD color gauge field
  static constexpr int TensorWords{ 4 * 3 * 3 * 2 }; // Lorentz * color matrix * complex
  using Tensor = double[TensorWords];
  static constexpr int TensorSize{ sizeof( Tensor ) };

  static constexpr int root{ 0 }; // MPI root node on COMM_WORLD
  static constexpr std::size_t GB2{ 1ul << 31 };

  int getWorldSize() const;
  int getWorldRank() const;

  bool ValidateMPIDims();
  int nDims;
  std::vector<int> Strides;
  std::vector<int> Starts;

  bool ValidateGlobalDims();
  std::vector<int> GlobalStrides;
  std::vector<int> GlobalStarts;
  int              GlobalSites;
  std::size_t      GlobalWords;
  std::size_t      GlobalSize;
  std::vector<int> LocalDims;
  std::vector<int> LocalStrides;
  std::vector<int> LocalStarts;
  int              LocalSites;
  std::size_t      LocalWords;
  std::size_t      LocalSize;

  bool CreateTestFile( const char * FileName, long HeaderBytes );
};

/*
 Reproduce MPI error when reading > 2GB to a single rank

 Required argument 1: Filename used for destructive test
 Optional argument 2: Size of header in bytes. Default 0
 Optional argument 3:    MPI dimensions. Default: world_size,1,1,1
 Optional argument 4: Global dimensions. Default: 48,48,48,96
 NB: arguments 3 and four can be n-dimensional, but must match

 Fails:
  mpirun --mca io romio321 -np 2 MPIRead32 a.out 0 2.1 2304.4608
  mpirun --mca io romio321 -np 2 MPIRead32 a.out 0 2.1 4608.2304
 
 Succeeds:
  mpirun --mca io romio321 -np 2 MPIRead32 a.out 0 1.2 2304.4608
  mpirun --mca io romio321 -np 2 MPIRead32 a.out 0 1.2 4608.2304
  mpirun --mca io ompio    -np 2 MPIRead32 a.out 0 2.1 2304.4608
  mpirun --mca io ompio    -np 2 MPIRead32 a.out 0 2.1 4608.2304
  mpirun --mca io ompio    -np 2 MPIRead32 a.out 0 1.2 2304.4608
  mpirun --mca io ompio    -np 2 MPIRead32 a.out 0 1.2 4608.2304

 Hint:
  Replacing both occurrences of MPI_ORDER_FORTRAN with MPI_ORDER_C
  causes the success / fail MPI ordering with romio321 to reverse
 */

int main(int argc, const char * argv[])
{
  // First parameter is filename
  int iReturn = EXIT_FAILURE;
  if( argc >= 2 && argv[1][0] )
  {
    try
    {
      const char * FileName = argv[1];
      long HeaderBytes = 0;
      Reproducer r;
      r.MPIDims = { r.world_size, 1, 1, 1 };
      r.GlobalDims = { 48, 48, 48, 96 };
      if( argc >= 3 && ( HeaderBytes = FromString<int>( argv[2] ) ) < 0 )
        std::cout << "HeaderBytes must be > 0" << std::endl;
      else if( argc >= 4 && ( r.MPIDims = ArrayFromString<int>( argv[3] ) ).empty() )
        std::cout << "Bad MPI dimensions \"" << argv[3] << "\"" << std::endl;
      else if( argc >= 5 && ( r.GlobalDims = ArrayFromString<int>( argv[4] ) ).empty() )
        std::cout << "Bad global dimensions \"" << argv[4] << "\"" << std::endl;
      else
      {
        iReturn = EXIT_SUCCESS;
        r.Test( FileName, HeaderBytes );
      }
    }
    catch(const std::exception &e)
    {
      std::cerr << "Error: " << e.what() << std::endl;
      iReturn = EXIT_FAILURE;
    } catch( ... ) {
      std::cerr << "Error: Unknown exception" << std::endl;
      iReturn = EXIT_FAILURE;
    }
  }
  if( iReturn != EXIT_SUCCESS )
    std::cout << "Usage: MPIRead32 Filename [Header_bytes [MPI_dims [global_dims]]]" << std::endl;
  return 0; // iReturn;
}

/*
 This is the important part of the test
 */

void Reproducer::Test( const char * FileName, long HeaderBytes )
{
  // So we can break on following line when debugging
  assert( !MPI_Barrier( MPI_COMM_WORLD ) );
  if( ValidateMPIDims() && ValidateGlobalDims() )
  {
    // Define an MPI type for my site tensor
    MPI_Datatype TypeTensor;
    assert( !MPI_Type_contiguous( TensorWords, MPI_DOUBLE, &TypeTensor ) );
    assert( !MPI_Type_commit( &TypeTensor ) );
    // Define Global Layout
    std::cout << "Rank " << world_rank << " GDim " << GlobalDims << ", LDim " << LocalDims << ", GStarts " << GlobalStarts << std::endl;
    MPI_Datatype TypeGlobal;
    assert( !MPI_Type_create_subarray( nDims, &GlobalDims[0], &LocalDims[0], &GlobalStarts[0], MPI_ORDER_FORTRAN, TypeTensor, &TypeGlobal ) );
    assert( !MPI_Type_commit( &TypeGlobal ) );
    // Define Local Layout
    std::cout << "Rank " << world_rank << " LDim " << LocalDims << ", LDim " << LocalDims << ", LStarts " << LocalStarts << std::endl;
    MPI_Datatype TypeLocal;
    assert( !MPI_Type_create_subarray( nDims, &LocalDims[0], &LocalDims[0], &LocalStarts[0], MPI_ORDER_FORTRAN, TypeTensor, &TypeLocal ) );
    assert( !MPI_Type_commit( &TypeLocal ) );
    // Create test file if it doesn't exist or is wrong size
    if( CreateTestFile( FileName, HeaderBytes ) )
    {
      std::cout << std::fixed << std::setprecision( 0 ) << "Rank " << world_rank
                << " reading " << LocalSize << " bytes (" << LocalWords << " words)" << std::endl;
      MPI_File hFile;
      assert( !MPI_File_open( MPI_COMM_WORLD, FileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &hFile ) );
      assert( !MPI_File_set_view( hFile, HeaderBytes, TypeTensor, TypeGlobal, "native", MPI_INFO_NULL ) );
      std::vector<double> Buffer( LocalWords );
      MPI_Status Status;
      assert( !MPI_File_read_all( hFile, &Buffer[0], 1, TypeLocal, &Status) );
      std::cout << "Rank " << world_rank << " First=" << Buffer[0] << ", Last=" << Buffer.back() << std::endl;
      MPI_File_close( &hFile );
    }
    // Cleanup
    MPI_Type_free( &TypeLocal );
    MPI_Type_free( &TypeGlobal );
    MPI_Type_free( &TypeTensor );
  }
}

/*
 What follows is plumbing (MPI start/stop, test file construction, parameter parsing, etc)
 */

int Reproducer::getWorldSize() const
{
  // Initialize the MPI environment
  assert( !MPI_Init( NULL, NULL ) );
  // Get the number of processes
  int size;
  assert( !MPI_Comm_size( MPI_COMM_WORLD, &size ) );
  return size;
}

int Reproducer::getWorldRank() const
{
  // Get the rank of the process
  int rank;
  assert( !MPI_Comm_rank( MPI_COMM_WORLD, &rank ) );
  return rank;
}

bool Reproducer::ValidateMPIDims()
{
  // Prepare error message
  std::ostringstream s;
  s << "Rank " << world_rank << " MPI " << MPIDims;

  // Compute combined product size and compute strides
  nDims = static_cast<int>( MPIDims.size() );
  Strides.resize( nDims );
  int prod_dim{ nDims ? 1 : 0 };
  for( int i = 0; i < nDims; ++i )
  {
    if( MPIDims[i] <= 0 )
    {
      s << " invalid";
      std::cout << s.str() << std::endl;
      return false;
    }
    Strides[i] = prod_dim;
    prod_dim *= MPIDims[i];
  }
  if( prod_dim != world_size )
  {
    s << " product " << prod_dim << " != world_size " << world_size;
    std::cout << s.str() << std::endl;
    return false;
  }
  // Now convert my rank into a starting segment
  Starts.resize( nDims );
  prod_dim = world_rank;
  for( int i = nDims - 1; i >= 0; --i )
  {
    Starts[i] = prod_dim / Strides[i];
    prod_dim -= Starts[i] * Strides[i];
  }
  assert( !prod_dim && "Bug: Starts computed incorrectly" );
  s << ", Strides " << Strides << ", Starts " << Starts;
  std::cout << s.str() << std::endl;
  return true;
}

bool Reproducer::ValidateGlobalDims()
{
  // Prepare error message
  std::ostringstream s;
  s << "Rank " << world_rank << " Global dims " << GlobalDims;

  // Check that we match MPI dimensions
  if( GlobalDims.size() != nDims )
  {
    s << " in " << GlobalDims.size() << " dimensions doesn't match MPI in " << nDims << " dimensions";
    std::cout << s.str() << std::endl;
    return false;
  }

  // Compute Global and Local sites and strides, plus Local Dims
  GlobalStrides.resize( nDims );
  GlobalStarts.resize( nDims );
  LocalDims.resize( nDims );
  LocalStrides.resize( nDims );
  LocalStarts.resize( nDims );
  std::size_t prod_dim{ 1 };
  LocalSites = 1;
  for( int i = 0; i < nDims; ++i )
  {
    if( GlobalDims[i] <= 0 || GlobalDims[i] % MPIDims[i] )
    {
      s << " invalid";
      std::cout << s.str() << std::endl;
      return false;
    }
    GlobalStrides[i] = static_cast<int>( prod_dim );
    LocalDims[i] = GlobalDims[i] / MPIDims[i];
    LocalStrides[i] = LocalSites;
    GlobalStarts[i] = Starts[i] * LocalDims[i];
    LocalStarts[i] = 0;
    LocalSites *= LocalDims[i];
    prod_dim *= GlobalDims[i];
  }
  s << " GlobalSites " << prod_dim;
  if( prod_dim >= GB2 )
  {
    s << " >= 2 GB. Unable to perform test.";
    std::cout << s.str() << std::endl;
    return false;
  }
  // Compute lengths
  GlobalSites = static_cast<int>( prod_dim );
  GlobalWords = prod_dim * TensorWords;
  GlobalSize = prod_dim * TensorSize;
  LocalWords = LocalSites;
  LocalWords *= TensorWords;
  LocalSize = LocalSites;
  LocalSize *= TensorSize;
  s << " (words " << GlobalWords << "), local sites " << LocalSites << "\n  local bytes " << LocalSize;
  if( LocalSize < GB2 )
    std::cout << s.str() << " < 2GB. Issue should not occur." << std::endl;
  else
    std::cout << s.str() << " >= 2GB. Issue should occur." << std::endl;
  return true;
}

bool Reproducer::CreateTestFile( const char * FileName, long HeaderBytes )
{
  int iOK = 0;
  if( world_rank != root )
  {
    std::cout << "Rank " << world_rank << " waiting for test file" << std::endl;
    MPI_Bcast( &iOK, 1, MPI_INT, root, MPI_COMM_WORLD );
  }
  else
  {
    struct stat64 buf;
    iOK = ( !stat64( FileName, &buf ) && buf.st_size == ( GlobalSize + HeaderBytes ) ) ? 1 : 0;
    std::cout << "Rank " << world_rank << (iOK ? " reusing " : " creating ") << GlobalSize << " byte file with "
              << HeaderBytes << " byte header" << std::endl;
    if( !iOK )
    {
      std::fstream myfile( FileName, std::ios::out | std::ios::binary );
      bool bOK = myfile.is_open();
      if( bOK )
      {
        // Write the Header
        static const char Header[] = "The nth dword in the payload=n. "
	  "Reading this back allows us to check FORTRAN ordering, "
	  "but this won't be sequential for multidimensional types. ";
        static constexpr int HeaderSize{ sizeof( Header ) - 1 };
        for( long ToWrite = HeaderBytes; bOK && ToWrite > 0; )
        {
          long ThisLen{ ToWrite > HeaderSize ? HeaderSize : ToWrite };
          myfile.write( Header, ThisLen );
          bOK = !myfile.bad();
          ToWrite -= ThisLen;
        }
        // Now write the payload, i.e. a series of incrementing doubles
        Tensor Buffer;
        double d = 0;
        for( int Site = 0; bOK && Site < GlobalSites; ++Site )
        {
          for( int i = 0; i < TensorWords; ++i )
            Buffer[i] = d++;
          myfile.write( reinterpret_cast<char*>( &Buffer[0] ), TensorSize );
          bOK = !myfile.bad();
        }
        myfile.close();
      }
      iOK = bOK ? 1 : 0;
      if( !bOK )
        std::cout << "Rank " << world_rank << " file creation failed." << std::endl;
    }
    MPI_Bcast( &iOK, 1, MPI_INT, root, MPI_COMM_WORLD );
  }
  return iOK;
}

// Check that a stream is empty ... or only contains white space to end of stream
template <class _CharT, class _Traits> inline bool StreamEmpty( std::basic_istream<_CharT, _Traits> & s )
{
  return s.eof() || ( s >> std::ws && s.eof() );
}

// Check string can be converted to requested type with no leftovers
template<typename T> inline T FromString( const std::string &String )
{
  std::istringstream s{ String };
  T value;
  if( StreamEmpty( s ) || !( s >> value ) || !StreamEmpty( s ) )
    throw std::runtime_error( "Bad input \"" + String + "\"" );
  return value;
}

// Convert string to array of type. Stop on failure. ',', '_' space or '.' separator
template<typename T> inline std::vector<T> ArrayFromString( const std::string &String )
{
  std::vector<T> v;
  std::istringstream s{ String };
  T value;
  while( !StreamEmpty( s ) && s >> value )
  {
    v.push_back( value );
    if( !s.eof() && ( s.peek() == '.' || ( s >> std::ws && !s.eof() && ( s.peek() == ',' || s.peek() == '_' ) ) ) )
      s.get();
  }
  return v;
}

// Display array using '.' as separator
template<typename T> std::ostream & operator<<( std::ostream &s, const std::vector<T> &v )
{
  for( std::size_t i = 0; i < v.size(); ++i )
  {
    if( i ) s << ".";
    s << v[i];
  }
  return s;
}
